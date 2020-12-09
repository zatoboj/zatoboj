cd /content/gdrive/My\ Drive/ybshmmlchk/zatoboj
mkdir -p ~/.ssh
cp ../id_rsa ~/.ssh
ssh-keyscan -t rsa github.com >> /root/.ssh/known_hosts
ssh -T git@github.com
source ../git_config_startup.sh
git pull