class BaseBERT(pl.LightningModule):
    def __init__(self, wrapped_config):
        super(BaseBERT, self).__init__()    
        # initializing parameters
        self.config = wrapped_config[0]
        self.batch_size = self.config.model.batch_size     
        self.max_len = self.config.model.max_len
        self.freeze_layers = self.config.model.freeze_layers
        self.lr = self.config.model.lr
        # save hyperparameters for .hparams attribute
        self.save_hyperparameters()
        # initializing BERT
        self.bert = get_transformer(self.config).cuda()
        self.bert_dim = self.bert.config.hidden_size
        # evaluation metrics
        self.val_metrics = ['plain', 'bysum', 'byend']
        # initializing dataloaders
        self.squad_train_dataloader, self.squad_val_dataloader, self.squad_test_dataloader = generate_squad_dataloaders(self.config)
        # initializing additional layers -- start and end vectors
        self.Start = nn.Linear(self.bert_dim, 1)
        self.End = nn.Linear(self.bert_dim, 1)
        self.custom_step = 0
        
    def new_layers(self, bert_output, new_layer):
        logits_wrong_shape = new_layer(torch.reshape(bert_output, (bert_output.shape[0]*bert_output.shape[1], bert_output.shape[2])))
        logits = torch.reshape(logits_wrong_shape, (bert_output.shape[0], bert_output.shape[1]))
        return logits

    def forward(self, batch):
        input_ids, attention_mask, token_type_ids, _, _, _, _, _ = batch
        # _ should be used for classification answer/no answer
        bert_output, _ = self.bert(input_ids=input_ids, 
                         attention_mask=attention_mask, 
                         token_type_ids=token_type_ids)
        # shape of q will be (batch_size, max_len, bert_dim) = (batch_size, 256, 768)
        # take inner products of output vectors with trainable start and end vectors
        start_logits = self.new_layers(bert_output, self.Start)
        end_logits = self.new_layers(bert_output, self.End)

        return start_logits, end_logits

    # this is the main function of pl modules. defines architecture and loss function. training loop comes for free -- implemented inside PL
    def training_step(self, batch, batch_nb):
        predictions = self.forward(batch)     
        loss = self.compute_loss(predictions, batch)

        self.custom_step += batch[0].shape[0]
        # logs
        self.logger.experiment.log({
            'train_loss' : loss,
            'epoch' : self.current_epoch
            }, step = self.custom_step)

        return {'loss': loss}

    def compute_loss(self, predictions, batch):
        start_logits, end_logits = predictions
        # LOSS: compute cross_entropy loss between predictions and actual labels for start and end 
        _, _, _, _, _, _, answer_starts, answer_ends = batch
        start_loss = F.cross_entropy(start_logits, answer_starts)
        end_loss = F.cross_entropy(end_logits, answer_ends)
        loss = start_loss + end_loss
        return loss

    def validation_step(self, batch, batch_nb):
        evaluator = Evaluator(self)
        _, val_dict = evaluator.evaluate_on_batch(batch) 
        return val_dict

    def validation_epoch_end(self, val_step_outputs):
        log_dict = {}
        for key in val_step_outputs[0]:
            aggregated = np.mean([accuracy_dict[key] for accuracy_dict in val_step_outputs])
            log_dict[key] = aggregated
        self.logger.experiment.log(log_dict, step = self.custom_step)
        self.log('val_loss', log_dict['val_loss'], prog_bar=True, logger=False)
        # delete models from Trash using pydrive
        if self.config.dirs.py_drive:
            for a_file in self.config.dirs.py_drive.ListFile({'q': "trashed=true"}).GetList():
                if a_file['title'] in {'model.ckpt', 'model-v0.ckpt'}:
                    title = a_file['title']
                    a_file.Delete()
                    print(f'File {title} was deleted from Trash.')
                  
    def configure_optimizers(self):
        return torch.optim.Adam([p for p in self.parameters() if p.requires_grad], lr=self.lr, eps=1e-08)

    def get_predictions(self, batch):
        '''
        Returns arrays (start probabilities, end probabilities) on given batch 
        '''    
        with torch.no_grad():
            start_prob, end_prob = self.forward(batch)
        return start_prob, end_prob

    def convert_predictions(self, predictions, min_start, metric='plain'):
        '''
        Return numpy arrays of predictions of indices of starts and ends for:
        - metric='plain' - as argmax of unnormalized probability vectors
        - metric='bysum' - as argmax of the sum of unrromalized probabilities over all pairs (i,j) such that i<j (and i>min_start if given)
        - metric='byend' - as argmax of unrromalized probabilities over all i>min_start for end and
                        as argmax of unrromalized probabilities over all min_start<j<end_pred for start   
        '''
        start_prob, end_prob = predictions
        neg_inf = -100
        batch_size, max_len = start_prob.shape
        if metric == 'plain':
            start_pred = np.argmax(start_prob, axis=1)
            end_pred = np.argmax(end_prob, axis=1)       
        elif metric == 'bysum':
            probs = start_prob.reshape(-1,max_len,1) + end_prob.reshape(-1,1,max_len) # array of shape: (batch_size, max_len, max_len), matrix of pairwise sums per each element of the batch
            mask = np.zeros(probs.shape)  # create a mask to avoid including cases where i > j or i > min_start or j > min_start
            for i,s in enumerate(min_start):
                mask[i,:s,:] = 1
                mask[i,:,:s] = 1
                mask[i][np.tril_indices(max_len,-1)] = 1
            mask[:,0,0] = 0               # we however leave i=j=0 to detect questions without answers
            probs = np.ma.array(probs,mask=mask)
            probs = np.ma.filled(probs,neg_inf)
            max_probs = np.argmax(probs.reshape(batch_size,-1), axis=-1) # array of shape: (batch_size,), argmaxes of flattened matrices of pairwise sums
            start_pred, end_pred = np.unravel_index(max_probs, (max_len, max_len)) # two arrays of shape: (batch_size,), 'unflattenning' of max_probs
        elif metric == 'byend':
            # first we deal with ends
            mask = np.zeros(end_prob.shape)  # create a mask to avoid including cases where end > min_start
            for i,s in enumerate(min_start):
                mask[i,:s] = 1
            mask[:,0] = 0               # we however leave end=0 to detect questions without answers
            end_prob = np.ma.array(end_prob,mask=mask)
            start_prob = np.ma.array(start_prob,mask=mask)
            end_prob = np.ma.filled(end_prob,neg_inf)
            start_prob = np.ma.filled(start_prob,neg_inf)
            end_pred = np.argmax(end_prob, axis=-1) # array of shape: (batch_size,), argmaxes of ends' probabilities
            # now we deal with starts
            mask = np.zeros(start_prob.shape)  # create a mask to avoid including cases where end > min_start
            for i,e in enumerate(end_pred):
                mask[i,e+1:] = 1
            start_prob = np.ma.array(start_prob,mask=mask)
            start_prob = np.ma.filled(start_prob,neg_inf)
            start_pred = np.argmax(start_prob, axis=-1) # array of shape: (batch_size,), argmaxes of starts' probabilities
        return start_pred, end_pred

    def train_dataloader(self):
        return self.squad_train_dataloader

    def val_dataloader(self):
        return self.squad_val_dataloader

    def test_dataloader(self):
        return self.squad_test_dataloader