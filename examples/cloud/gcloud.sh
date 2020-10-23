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


filedir="201007"
gsutil -o 'GSUtil:parallel_process_count=1' -o 'GSUtil:parallel_thread_count=16' -m cp -r "gs://active-ird-experiments/rss-logs/logs/output/$filedir/*" "data/$filedir/"

expdir="active_ird_ibeta_50_w1_joint_dbeta_1_dvar_1_prior_0_eval_mean_128_seed_0_603_adam"
filedir="200705"
gsutil -o 'GSUtil:parallel_process_count=1' -o 'GSUtil:parallel_thread_count=16' -m cp -r "gs://active-ird-experiments/rss-logs/logs/output/$filedir/$expdir*" "data/$filedir/"


filedir="200930"
expdir="active_ird_simplified_indep_init_1v3_ibeta_6_obs_true_dbeta_0.1"
gsutil -o 'GSUtil:parallel_process_count=1' -o 'GSUtil:parallel_thread_count=16' -m cp -r "gs://active-ird-experiments/rss-logs/logs/output/$filedir/$expdir*" "data/$filedir/"



filedir="201012"
expdir="iterative_divide_initial_1v1_*"
gsutil -o 'GSUtil:parallel_process_count=1' -o 'GSUtil:parallel_thread_count=16' -m cp -r "gs://active-ird-experiments/rss-logs/logs/output/$filedir/$expdir" "data/$filedir/"

gsutil -o 'GSUtil:parallel_process_count=1' -o 'GSUtil:parallel_thread_count=16' -m ls -r "gs://active-ird-experiments/rss-logs/logs/output/$filedir/$expdir"


## SSH
gcloud compute ssh hzyjerry@user-study-icra20-1


## Copy local to remote

expdir="iterative_divide_initial_1v1_icra_20"
# expdir="iterative_divide_initial_1v1_icra_10"
# zone="us-central1-a"
zone="us-west2-a"
# remote='user-study-icra20-1'
remote='user-study-icra20-4'
gcloud compute scp --zone "$zone" --recurse "data/201012/$expdir" "hzyjerry@$remote:~/study/rdb/examples/notebook/"



## Download remote data from notebook folder
expdir="iterative_divide_initial_1v1_icra_20"
# zone="us-central1-a"
zone="us-west2-a"
# remote='user-study-icra20-4'
remote='user-study-icra20-7'
# remote='user-study-icra20-1'
gcloud compute scp --zone "$zone" --recurse "hzyjerry@$remote:~/study/rdb/examples/notebook/$expdir" "data/201012/" && gcloud compute scp --zone "$zone" --recurse "hzyjerry@$remote:~/study/rdb/data/201012/$expdir" "data/201012/"



## Download remote data from data folder
expdir="iterative_divide_initial_1v1_icra_08"
# zone="us-central1-a"
zone="us-west2-a"
# remote='user-study-icra20-1'
# remote='user-study-icra20-4'
remote='user-study-icra20-7'
gcloud compute scp --zone "$zone" --recurse "hzyjerry@$remote:~/study/rdb/data/201012/$expdir" "data/201012/"



gcloud compute scp --zone us-central1-a --recurse hzyjerry@user-study-icra20-1:~/study/rdb/data/201012/iterative_divide_initial_* data/201012




gcloud compute scp --zone us-central1-a --recurse hzyjerry@user-study-icra20-1:~/study/rdb/data/201012/iterative_divide_initial_*_icra_06 data/201012

gcloud compute scp --zone us-central1-a --recurse hzyjerry@user-study-icra20-1:~/study/rdb/data/201012/iterative_divide_initial_*_icra_06 data/201012

gcloud compute scp --zone us-west2-a --recurse hzyjerry@user-study-icra20-7:~/study/rdb/data/201012/iterative_divide_initial_*_icra_06 data/201012

gcloud compute scp --zone us-west2-a --recurse hzyjerry@user-study-icra20-7:~/study/rdb/data/201012/iterative_divide_initial_1v3_icra_01 data/201012

screen

cd study/rdb/examples/notebook && source activate studyenv
xvfb-run -s "-screen 0 1400x900x24" jupyter-notebook --no-browser --port=5000 --NotebookApp.token=abcd
