# Setup
git clone git@github.com:baker-laboratory/rf_diffusion.git

cd rf_diffusion

git submodule init

git submodule update --init

## Temporary hack
export PYTHONPATH="${PYTHONPATH}:/home/ahern/projects/aa/rf_diffusion_flow/lib/se3_flow_matching"

cd rf_diffusion

## Verify tests pass
pytest --disable-warnings -s
