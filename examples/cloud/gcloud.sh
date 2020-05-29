## Check logs
gsutil ls -lh "gs://active-ird-experiments/rss-logs/" | sort -k 3


## SSH
# gcloud beta compute --project "aerial-citron-264318" ssh --zone "us-west1-b" "active-ird-00"

## Make public
# gsutil acl ch -u AllUsers:R "gs://active-ird-experiments/rss-logs/logs/input/200110_test_eval_all.tar.gz"
# https://storage.googleapis.com/active-ird-experiments/rss-logs/logs/input/200110_test_eval_all.tar.gz


## Remove dar mount
# gsutil -m rm gs://active-ird-experiments/doodad/mount/*


## Retrieve files
filedir="200322"
expdir="active_ird_sum_ibeta_50_dprior_2_dbeta_20_obs_uniform_w1_unif_128_602_adam"
mkdir "data/$filedir/$expdir"
gsutil -o 'GSUtil:parallel_process_count=1' -o 'GSUtil:parallel_thread_count=16' -m cp -r "gs://active-ird-experiments/rss-logs/logs/output/$filedir/$expdir/*" "data/$filedir/$expdir/"



filedir="200410"
gsutil -o 'GSUtil:parallel_process_count=1' -o 'GSUtil:parallel_thread_count=16' -m cp -r "gs://active-ird-experiments/rss-logs/logs/output/$filedir/*" "data/$filedir/"


filedir="200528"
gsutil -o 'GSUtil:parallel_process_count=1' -o 'GSUtil:parallel_thread_count=16' -m cp -r "gs://active-ird-experiments/rss-logs/logs/output/$filedir/*" "data/$filedir/"

filedir="200528"
expdir="active_ird_ibeta_50_w1_joint_dbeta_1_dvar_1_prior_0_eval_mean_128_seed_0_603_adam"
gsutil -o 'GSUtil:parallel_process_count=1' -o 'GSUtil:parallel_thread_count=16' -m cp -r "gs://active-ird-experiments/rss-logs/logs/output/$filedir/$expdir*" "data/$filedir/"


filedir="200515"
gsutil -o 'GSUtil:parallel_process_count=1' -o 'GSUtil:parallel_thread_count=16' -m cp -r "gs://active-ird-experiments/rss-logs/logs/output/$filedir/iterative_divide/*.npy" "data/$filedir/iterative_divide/"
