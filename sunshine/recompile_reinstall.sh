# 编译
apt remove -y sunshine
apt install -y libappindicator3-1
./scripts/linux_build.sh
# 安装
apt install -y libayatana-appindicator3-1
apt --fix-broken -y install
dpkg -i build/cpack_artifacts/Sunshine.deb
