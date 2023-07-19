INSTRUCTION=$1 # all or features or xvectors or VBx or score
SET=$2 # dev or eval or inference
DIHARD_DIR=$3 # directory containing the DIHARD data as provided by the organizers

if [[ $SET = "dev" ]]; then
	DATA_DIR=$DIHARD_DIR #/LDC2019E31_Second_DIHARD_Challenge_Development_Data
	VBX_DIR=.
elif [[ $SET = "eval" ]]; then
	DATA_DIR=$DIHARD_DIR #/LDC2019E32_Second_DIHARD_Challenge_Evaluation_Data_V1.1
	VBX_DIR=.
elif [[ $SET = "inference" ]]; then
	#DATA_DIR=$DIHARD_DIR #custom set for ensemble model
	AUDIO_DIR=$3
	VBX_DIR=VBx

else
	echo "The set has to be 'dev' or 'eval' or 'inference'"
	exit -1
fi

TMP_DIR=./tmp_dir_$SET
OUT_DIR=./out_dir_$SET
if [[ $SET = "inference" ]]; then
	OUT_DIR=$4
	BN_DIR=$5
	TMP_DIR=VBx/tmp_dir_$BN_DIR
fi

mkdir -p $TMP_DIR $OUT_DIR

if [[ $INSTRUCTION = "all" ]] || [[ $INSTRUCTION = "features" ]]; then
	# For recordings listed in file $1
	# - load lab file with VAD information from directory $2
	# - load original or WPE processed flac file from directory $3
	# - extract speech segment according to VAD
	# - each speech segment split into 1.5s subsegments with shift 0.25s
	# - extract Kaldi compatible cmv normalized fbank features for each subsegment
	# - save all subsegment feature matrices to ark file $4
	# - save subsegment timing information to $5
	if [[ $SET = "inference" ]]; then
		VBx/compute_fbanks_cmn.py \
			VBx/inference_files/list_inference_$BN_DIR \
			$OUT_DIR \
			$AUDIO_DIR \
			$TMP_DIR/fbank_cmn.ark \
			$TMP_DIR/segments
	else
		./compute_fbanks_cmn.py \
		  list_$SET \
		  $DATA_DIR/data/pred_sad \
		  $DATA_DIR/data/flac \
		  $TMP_DIR/fbank_cmn.ark \
		  $TMP_DIR/segments
	fi
fi


if [[ $INSTRUCTION = "all" ]] || [[ $INSTRUCTION = "xvectors" ]]; then
	feats_ark=$TMP_DIR/fbank_cmn.ark
	model_init=$VBX_DIR/xvector_extractor.txt
	feats_len=`wc -l $TMP_DIR/segments | awk '{print $1}'`
	arkfile=$TMP_DIR/xvectors.ark

	$VBX_DIR/extract.py --feats-ark $feats_ark \
                 --feats-len $feats_len \
                 --ark-file $arkfile \
                 --batch-size 1 \
                 --model-init $model_init
fi


if [[ $INSTRUCTION = "all" ]] || [[ $INSTRUCTION = "VBx" ]]; then
	alpha=0.55
	thr=0.0
	tareng=0.3
	smooth=5.0
	lda_dim=220
	Fa=0.4
	Fb=11
	loopP=0.80

	# x-vector clustering using VBHMM based diarization
	$VBX_DIR/diarization_PLDAadapt_AHCxvec_BHMMxvec.py \
	 					$OUT_DIR \
	 					$TMP_DIR/xvectors.ark \
	 					$TMP_DIR/segments \
	 					$VBX_DIR/mean.vec \
	 					$VBX_DIR/transform.mat \
	 					$VBX_DIR/plda_voxceleb \
	 					$VBX_DIR/plda_dihard \
	 					$alpha \
	 					$thr \
	 					$tareng \
	 					$smooth \
	 					$lda_dim \
	 					$Fa \
	 					$Fb \
	 					$loopP
fi


# if [[ $INSTRUCTION = "all" ]] || [[ $INSTRUCTION = "score" ]]; then
# 	SCORE_DIR=$4 # directory with scoring tool: https://github.com/nryant/dscore
# 	python $SCORE_DIR/score.py \
# 		--collar 0.0 \
# 		-r $DATA_DIR/data/rttm/*.rttm \
# 		-s $OUT_DIR/*.rttm
# fi
