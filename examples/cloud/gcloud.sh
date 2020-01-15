gcloud beta compute --project "aerial-citron-264318" ssh --zone "us-west1-b" "active-ird-00"

# Make public
gsutil acl ch -u AllUsers:R "gs://active-ird-experiments/rss-logs/logs/input/200110_test_eval_all.tar.gz"
# https://storage.googleapis.com/active-ird-experiments/rss-logs/logs/input/200110_test_eval_all.tar.gz
