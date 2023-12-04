# Setup
git clone git@github.com:baker-laboratory/rf_diffusion.git PKG_DIR

cd rf_diffusion

git submodule init

git submodule update --init

## Temporary hack
export PYTHONPATH="${PYTHONPATH}:PKG_DIR:PKG_DIR/lib/se3_flow_matching"

cd rf_diffusion

## Verify tests pass
pytest --disable-warnings -s
