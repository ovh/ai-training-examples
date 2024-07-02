# data config
GROUP      ?= experiments
MODEL      ?= lego
OUTPUT_DIR := logs/$(GROUP)/$(MODEL)

# preprocess config
DOWNSAMPLE_RATE ?= 2
SCENE_TYPE      ?= object

# S3 config
DATASTORE   ?= NEURALANGELO
BUCKET_NAME := neuralangelo-$(shell whoami)-$(MODEL)
BUCKET      := $(BUCKET_NAME)@$(DATASTORE)

# job config
FLAVOR ?= ai1-1-gpu
GPU    ?= 1
VOLUME := $(BUCKET)/neuralangelo:/neuralangelo:rw:cache

# process config
ITERATIONS ?= 1000

# extraction config
RESOLUTION ?= 2048
BLOCK_RES  ?= 128

logs/:
	@mkdir -p logs

neuralangelo/:
	git clone https://github.com/NVlabs/neuralangelo.git
	git -C neuralangelo submodule update --init --recursive

BlenderNeuralangelo/:
	git clone https://github.com/mli0603/BlenderNeuralangelo

.PHONY: prepare
prepare logs/prepare_job.json: | neuralangelo/ logs/
	ovhai job run \
		-o json \
		-g $(GPU) \
		-f $(FLAVOR) \
		-v $(VOLUME) \
		docker.io/chenhsuanlin/colmap:3.8 -- \
      		bash -c "cd /neuralangelo && \
			bash projects/neuralangelo/scripts/preprocess.sh $(MODEL) input/$(MODEL).mp4 $(DOWNSAMPLE_RATE) $(SCENE_TYPE)" \
		> logs/prepare_job.json

.PHONY: adjust
adjust: BlenderNeuralangelo/
	blender --python BlenderNeuralangelo/start_blender_with_addon.py

.PHONY: process
process logs/process_job.json: | neuralangelo/ logs/
	ovhai job run \
		-o json \
		-g $(GPU) \
		-f $(FLAVOR) \
		-v $(VOLUME) \
		docker.io/chenhsuanlin/neuralangelo:23.04-py3 -- \
      		bash -c "cd /neuralangelo && \
			torchrun --nproc_per_node=$(GPU) train.py \
				--logdir=$(OUTPUT_DIR) \
				--show_pbar \
				--config=projects/neuralangelo/configs/custom/$(MODEL).yaml \
				--max_iter=$(ITERATIONS)" \
		> logs/process_job.json

neuralangelo/$(OUTPUT_DIR)/latest_checkpoint.txt:
	ovhai bucket object download $(BUCKET) $@

.PHONY: extract
extract logs/extract_job.json: neuralangelo/$(OUTPUT_DIR)/latest_checkpoint.txt | neuralangelo/ logs/
	ovhai job run \
		-o json \
		-g $(GPU) \
		-f $(FLAVOR) \
		-v $(VOLUME) \
		docker.io/chenhsuanlin/neuralangelo:23.04-py3 -- \
      		bash -c "cd /neuralangelo && \
		 	torchrun --nproc_per_node=$(GPU) projects/neuralangelo/scripts/extract_mesh.py \
				--config=$(OUTPUT_DIR)/config.yaml \
				--checkpoint=$(OUTPUT_DIR)/$(shell cat $<) \
				--output_file=$(OUTPUT_DIR)/$(MODEL).ply \
				--resolution=$(RESOLUTION) --block_res=$(BLOCK_RES) \
				--textured" \
		> logs/extract_job.json

.PHONY: prepare-job process-job extract-job
prepare-job process-job extract-job: %-job: | logs/%_job.json
	ovhai job get $(shell cat logs/$*_job.json | jq -r .id)

.PHONY: prepare-status process-status extract-status
prepare-status process-status extract-status: %-status: | logs/%_job.json
	@ovhai job get $(shell cat logs/$*_job.json | jq -r .id) -o json | jq -r .status.state

.PHONY: prepare-logs process-logs extract-logs
prepare-logs process-logs extract-logs: %-logs: | logs/%_job.json
	ovhai job logs -f $(shell cat logs/$*_job.json | jq -r .id)

.PHONY: clear-data
clear-data:
	ovhai bucket object rm $(BUCKET) --all -y

.PHONY: push-data
push-data: | neuralangelo/
	ovhai bucket object upload $(BUCKET) .

.PHONY: pull-data
pull-data:
	ovhai bucket object download $(BUCKET)

.PHONY: clean
clean:
	rm -Rf neuralangelo/datasets/*/ neuralangelo/logs/ logs/
