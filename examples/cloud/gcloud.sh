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
filedir="200206"
gsutil -o 'GSUtil:parallel_process_count=1' -o 'GSUtil:parallel_thread_count=16' -m cp -r "gs://active-ird-experiments/rss-logs/logs/output/$filedir/*" "data/$filedir/"
