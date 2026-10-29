#!/data/data/com.termux/files/usr/bin/sh
# RVN Fold4 one-shot W1 corpus build (raven_w1/run_fold4.sh)
H=$HOME
cd $H || exit 1

echo STEP1 build script
curl -sL "https://raw.githubusercontent.com/mawangsung/cicada/vibe/raven-w1-runner/raven_w1/raven_w1_build_script_r2.py" -o raven_w1_build_script_r2.py
if [ ! -s raven_w1_build_script_r2.py ]; then echo FAIL1; exit 1; fi
head -1 raven_w1_build_script_r2.py

echo STEP2 sqlite
if [ -s $H/chatgpt.sqlite ]; then
echo sqlite exists
else
echo downloading 853MB, rerun if interrupted
curl -L -C - "https://drive.usercontent.google.com/download?id=1E17ZkDYM59_muDsb_OjD69Y6BoJ06F8R&export=download&confirm=t" -o $H/chatgpt.sqlite
fi
if [ ! -s $H/chatgpt.sqlite ]; then echo FAIL2; exit 1; fi

echo STEP3 integrity
python3 -c "import sqlite3; r=sqlite3.connect('file:'+'$H'+'/chatgpt.sqlite?mode=ro',uri=True).execute('PRAGMA integrity_check').fetchone()[0]; print('integrity',r)"

echo STEP4 build
python3 raven_w1_build_script_r2.py $H/chatgpt.sqlite $H/raven_w1_corpus_r1 > $H/w1_build_log.txt 2>&1
tail -8 $H/w1_build_log.txt

echo STEP5 pack
if [ -d $H/raven_w1_corpus_r1 ]; then
tar -czf $H/w1_result.tar.gz -C $H raven_w1_corpus_r1 w1_build_log.txt
echo DONE
else
echo BUILD FAILED, see w1_build_log.txt
fi
