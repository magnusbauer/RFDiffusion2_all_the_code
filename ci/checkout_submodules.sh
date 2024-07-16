# This script is used to checkout the submodules of the main repository.
# It is necessary because github/baker and git.ipd need different tokens, and
# the standard github action doesn't seem to support this. booooooo.
# The script is called by the CI pipeline and should not be called manually.

ACT=$1 # if in local testing environment
GITLAB_SHEFFLER=$2
BAKER_RF_DIFFUSION_SHEFFLER=$3

echo ACT $ACT

if [[ ! -z $ACT ]]; then
  mkdir -p $HOME/bare_repos
  rsync -a lappy:rf/RF2-allatom.git/ $HOME/bare_repos/RF2-allatom.git/
  rsync -a lappy:rf/frame-flow.git/ $HOME/bare_repos/frame-flow.git/
  rsync -a lappy:rf/fused_mpnn.git/ $HOME/bare_repos/fused_mpnn.git/
  ls $HOME/bare_repos/*.git
  ls $HOME/bare_repos/RF2-allatom.git
fi

echo RUNNER IN $(pwd)
git reset --hard

MPNN=$(git submodule status | head -n1 | cut -b2-40)
RF2AA=$(git submodule status | head -n2 | tail -n1 | cut -b2-40)
SE3=$(git submodule status | tail -n1 | cut -b2-40)
echo SUBMODULE COMMITS
echo rf2aa $RF2AA
echo se3_flow_matching $SE3
echo fused_mpnn $MPNN

cd lib && rm -rf rf2aa fused_mpnn se3_flow_matching

echo CHECKOUT_RF2
echo clone
git clone $HOME/bare_repos/RF2-allatom.git rf2aa
cd rf2aa
# git clone -b aa --reference RF2-allatom.git https://$GITLAB_SHEFFLER@git.ipd.uw.edu/jue/RF2-allatom.git rf2aa
if [[ -z $ACT ]]; then
  git remote set-url origin https://sheffler:$GITLAB_SHEFFLER@git.ipd.uw.edu/jue/RF2-allatom.git
else
  git remote set-url origin https://$GITLAB_SHEFFLER@git.ipd.uw.edu/jue/RF2-allatom.git
fi
echo fetch
git fetch
echo reset
git reset --hard $RF2AA
cd ..

echo CHECKOUT_MPNN
echo clone
mkdir fused_mpnn && cd fused_mpnn
git clone $HOME/bare_repos/fused_mpnn.git fused_mpnn
# git clone https://$BAKER_RF_DIFFUSION_SHEFFLER@github.com/baker-laboratory/fused_mpnn
git remote set-url origin https://$GITLAB_SHEFFLER@github.com/baker-laboratory/fused_mpnn
echo fetch
cd fused_mpnn && git fetch
echo reset
git reset --hard && git checkout $MPNN && cd ../..

echo CHECKOUT_SE3_FLOW
echo clone
git clone $HOME/bare_repos/frame-flow.git se3_flow_matching
# git clone https://$BAKER_RF_DIFFUSION_SHEFFLER@github.com/baker-laboratory/frame-flow se3_flow_matching
git remote set-url origin https://$BAKER_RF_DIFFUSION_SHEFFLER@github.com/baker-laboratory/frame-flow
echo reset
cd se3_flow_matching && git fetch
echo reset
git reset --hard && git checkout $SE3 && cd ..

cd ..
