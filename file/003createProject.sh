source /home/zhike/proxy_file
cd verify
rm -rf ./*

# git clone https://github.com/lizardbyte/sunshine.git --recurse-submodules
cp -rp ../sunshine-master sunshine/
cd sunshine
git reset --hard fb712e
git submodule update --init --recursive