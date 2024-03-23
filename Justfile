[private]
@default:
	just --list --unsorted

# Run the sample prompt against the model
test:
	python run.py

# Download the weights for the Grok model
download-weights:
	transmission-cli \
		--download-dir ./checkpoints \
		$GROK_MAGNET_LINK
	ln -s ./checkpoints/grok-1/ckpt-0 ./checkpoints/ckpt-0
