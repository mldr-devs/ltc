# Learning to communicate

## Julia

```
# w katalogu projektu
cd /Users/krzysiek/Documents/ml4wifi/ltc
source .venv/bin/activate

pip install -U pysr juliacall juliapkg

# usuń wygenerowane środowisko julia, juliapkg stworzy je ponownie
rm -rf .venv/julia_env

# uruchom ponownie moduł (juliapkg odbuduje środowisko)
python -m ltc.symbolic.nn2sym
```