# Training parameters and provenance

The per-value record is manifests/parameter_provenance.csv. It separates executed code, saved checkpoint configuration, revision protocol and unrecovered history.

Historical RETFound source checkpoints for seeds 20260906 and 20260908 save batch 24, accumulation 1, 30 maximum epochs, 5 warm-up epochs and patience 6. The actual source script computes learning rate as 5e-3 times effective batch / 256, giving 0.00046875. It computes image-level class weights from 1,392 negative and 410 positive BRSET training images: [1, 3.3951219512]. This is not a patient-level weight, and the source script default alone would not prove the saved execution configuration. Seed 20260907 has a remote hash but its saved local configuration was not recovered.

The executed historical VisionFM frozen decoder uses last-four-block CLS concatenation (3072 features), BatchNorm, 3072/1536/768/1 MLP, AdamW 0.001, weight decay 0.05, batch 256, cosine schedule, maximum 100 epochs and patience 15, selected by BRSET validation patient AUROC.

The revision R1-1 arm instead freezes official RETFound and trains a 1024/512/256/1 MLP with 658,433 trainable parameters. It uses the same head optimizer settings but has different pretraining, feature aggregation, preprocessing and absolute capacity. It is not a pure architecture experiment. Other revision adaptation runs require their exact per-run configurations and internal command ledger; one guessed common parameter set would be incorrect.
