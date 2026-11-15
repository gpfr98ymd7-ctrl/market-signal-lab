# Market Lab

A weekly market and stock forecasting job with an installable phone dashboard and an honest scorecard.

- **Every Friday evening** a GitHub Action downloads data, checks it, forecasts five markets (S&P 500, Nasdaq 100, FTSE 100, Euro Stoxx 50, Nikkei 225), ranks a list of stocks, and applies a market-risk scaler.
- **A monthly health check** tests the model the honest way (walk-forward: each year is predicted using only earlier data). The result decides how far probabilities are pulled toward base rates and how much the return ranges are widened. If the model has shown no edge, the page says so and quotes base rates instead of confident-looking noise.
- **Every forecast is logged** and scored once enough days have passed, so you build a real track record.
- **A news archive and company data build up alongside it.** Each week saves the free news headlines, a daily tone and attention reading from GDELT, and a snapshot of each stock's valuation, quality, growth and next earnings date. They are shown as context, and the app tests honestly whether either would ever have improved the forecasts (see "News and company data" below).
- **A weekly Growth watch and two paper portfolios.** Each week the app lists the 20 stocks with the most upside potential over the next 5 to 7 trading days, builds a growth-weighted portfolio from them and "sells and buys" it on paper, so you can see what it would have done. It is pretend money only: nothing in this project can place a real order. See "Growth watch and paper trading" below.
- **Global market drivers.** Interest rates, the dollar, oil and other commodities, credit, market fear and a few regions are tracked, shown on the Markets tab, and offered to the models only where a walk-forward test shows they clearly help (see "Global drivers" below).
- `docs/` is a small web app (a PWA). Open it on your phone and add it to your home screen and it behaves like an app.

Research tool only, not financial advice. Direction forecasts are weak signals; volatility and risk ranges are more reliable. The Trust tab shows which is which.

---

## Setup from your phone

You need the GitHub website (or app) and a Google Colab session. Nothing needs a computer.

### 1. Create the repository
1. On github.com, tap **New repository**.
2. Name it `market-signal-lab` (any name works, but keep it consistent below).
3. Tick **Add a README file** so the repository has a branch to upload into.
4. Choose **Public** or **Private**. Read the privacy note at the end first.

### 2. Make a short-lived upload token
1. GitHub > your profile picture > **Settings** > **Developer settings** > **Personal access tokens** > **Fine-grained tokens** > **Generate new token**.
2. Expiration: **7 days**.
3. Repository access: **Only select repositories**, and choose `market-signal-lab`.
4. Permissions, under Repository permissions, set **Contents: Read and write** and **Workflows: Read and write**.
5. Generate it and copy the token.

### 3. Upload the project with the deploy notebook
1. Open `deploy_to_github.ipynb` in Google Colab.
2. Tap **Runtime > Run all**. When asked, type your GitHub username, then paste the token (it is hidden as you type) and press Enter. If your repository is not called `market-signal-lab`, change `REPO` in the first cell. (A Colab secret named `GITHUB_TOKEN` also works but is not required, and the phone view of Colab has no Secrets panel.)
3. Every file should print `OK` or `kept`.
4. Go back to GitHub and delete the token (Settings > Developer settings > Fine-grained tokens), since it is no longer needed.

You can run the deploy notebook again later to receive updates. It never overwrites your forecast log, your results, or your edited `marketlab/config.py` (see "Updating" below).

### 4. Turn on the web page
Public repository: **Settings > Pages > Build and deployment > Deploy from a branch**, choose branch `main` and folder `/docs`, then Save. After a minute your app is at `https://YOUR-USERNAME.github.io/market-signal-lab/`. Until the first real run it shows clearly marked **demo data** so you can see how it looks.

Private repository: GitHub Pages is not available for private repositories on the free plan. Either make the repository public, or host the `docs` folder with Cloudflare Pages or Netlify, both of which can deploy from a private GitHub repository (output directory: `docs`). Check each service's current free limits.

### 5. Run the first forecast
1. In the repository, open the **Actions** tab. If GitHub asks you to enable workflows, enable them.
2. Choose **Weekly forecast** > **Run workflow**.
3. Wait for the green tick. The first run is the slowest (roughly 10 to 25 minutes) because it also runs the health check; later runs are faster. It commits the results automatically.
4. Open your app URL. The demo banner disappears when real data arrives.

### 6. Install it on your phone
- **iPhone (Safari):** Share button > **Add to Home Screen**.
- **Android (Chrome):** menu > **Install app** or **Add to Home screen**.

After that it runs every Friday at 21:30 UTC (after the US close) without you doing anything. Tap **Refresh** to load the latest.

---

## Optional extras

### A notification on your phone
1. Install the free **ntfy** app and subscribe to a topic name that is hard to guess (for example `marketlab-8f3k29x`). Anyone who knows the topic name can read it, so nothing sensitive is sent.
2. In the repository: **Settings > Secrets and variables > Actions > New repository secret**, name `NTFY_TOPIC`, value is your topic name.

You get a short summary each Friday (high priority if there are alerts, and tapping it opens the dashboard), plus an alert if a run fails.

### AI-written weekly commentary
Add a repository secret named `ANTHROPIC_API_KEY` (a key from console.anthropic.com, which is separate from a claude.ai subscription and billed per use). The dashboard then shows a short plain-English commentary on the Overview tab, written from that week's numbers and public headlines. It never changes a forecast, and without the secret the feature simply stays off.

---

## What the dashboard shows
- **Overview:** a one-paragraph summary, alerts, what changed since last run, markets at a glance, the optional commentary, and free public headlines (context only; the model does not read the news).
- **Markets:** for each index, the probability of rising over 21 trading days, crash risk (a fall of more than 5% below today's level) against its normal rate, expected volatility against current, the 5-95% return range, the regime, and the risk scaler's suggested exposure (0.25 to 1.00). Pills show whether each forecast has historically beaten chance.
- **Stocks:** the probability of beating the universe average over 5 trading days, an illustrative top-5 portfolio (inverse-volatility weights multiplied by the home market's risk scaler, the rest in cash), and the turnover cost since last run. Stocks are assumed bought one trading day after the signal.
- **News tone (Overview):** the average tone and amount of English-language news coverage (GDELT) over the last 7 days for global markets and each index, compared with that series' own past year, with a 60-day chart. "Darker than usual" describes the news, not a forecast.
- **Company data (Stocks):** under each stock, tap **Company data** for P/E, return on equity, margins, growth, debt and next earnings date, plus a simple 0-100 rank against the other tracked stocks. A yellow chip appears when a top pick reports earnings within a week, because results can move a stock sharply and the model does not know about them. Tap **News check** for the stock's recent news tone and headlines.
- **Growth:** this week's top 20, each with a plain-English reason, its weight in the paper portfolio, what a global driver might do to it, and a news check; the trades the paper portfolio would make this week; the paper track record against the S&P 500 and an equal-weight basket; and how concentrated the list is.
- **Global drivers (Markets):** the latest reading for each driver, how unusual its recent move is, and whether the model is currently allowed to use the extra inputs.
- **Trust:** the walk-forward results behind every pill, a stricter test where calibration is learned from earlier years only, the risk scaler against buy-and-hold, the stock ranker against an equal-weight universe, your live scorecard, the paper-trading results and the global-driver test, and the known limits. Wait for about 26 weekly runs before reading much into the live scorecard.

## News and company data
Both are **context only**. Neither changes a forecast, a ranking or a weight. They exist so the app can find out, with evidence, whether they are worth using.

**News.** Every run it (1) saves new headlines from the free feeds to `data/news_headlines.csv`, with the date first seen; (2) merges GDELT's daily tone and attention series into `data/news_daily.csv`; and (3) once a month tests whether adding the news series to the market models would have improved them.
- GDELT's search normally reaches back about 3 months, so the app starts with roughly 3 months of tone history and grows it every week. It also tries a longer window about once a month in case GDELT allows it. If GDELT sends back something unusable, the News tone card on the Overview tab says exactly what it received.
- The test needs about 4 years (1,100 trading days) of history before it can run, so on a normal start it will say "not testable yet" for a long time. That is the honest position, not a bug.
- When it does run, it compares each result with 9 placebos (the same news moved to unrelated dates), so a lucky improvement from extra columns is not mistaken for evidence. In tests on synthetic data, pure random noise was called a help in about 7% of comparisons and never "promising"; without the placebos it was about half.

**Company data.** Yahoo only gives today's ratios, not what they were years ago, so training a model on them would quietly use the future. Instead each week's ratios are saved (`data/fundamentals.csv`), and after 24 weekly snapshots plus 21 trading days the Trust tab reports whether stocks with a higher company-data rank went on to beat lower-ranked ones. Until then it says "too early".
- The rank is deliberately simple (value, quality and growth percentile ranks, averaged) and is not adjusted for sector, so banks, for example, are compared with technology companies.
- Free cash flow is dropped when a company reports in a different currency from the one it trades in, and absurd P/E ratios are ignored.
- If Yahoo or GDELT refuses requests, the run carries on without that part and says so on the page. It never stops a forecast.

Turn either off with `NEWS_ENABLED = False` or `FUND_ENABLED = False` in `marketlab/config.py`, or run `python run_weekly.py --no-news --no-fundamentals`. Other settings (search phrases, time limits) are in `marketlab/opts.py`; add a line with the same name to `config.py` to override one.

## Growth watch and paper trading
**What it picks.** Every Friday the app scores each stock it follows by blending four ranks: the stock ranker's raw score (30%), business growth from the weekly company-data snapshot (30%), how far the stock typically moves in a week (25%), and company quality (15%). The top 20 are the Growth watch. The weights are an opinion, not a fitted result, and the stock ranker has so far shown no proven edge, so treat the list as a list of stocks with room to move in either direction, not as a forecast.

**The paper portfolio.** The 20 stocks are weighted towards the higher scores (half split equally, half in proportion to the score, no stock above 10%), "bought" at the close one trading day after the signal, and swapped for the next week's list a week later. Costs of 0.10% are charged on everything that changes between weeks. The week in progress is marked to market; a closed week is written once to `data/paper_growth.csv` and never changed.

**How it is judged.** Results are compared with the S&P 500 and with an equal-weight basket of every stock the app followed that week (the fair test of whether picking helped). No verdict is shown until 26 weeks have closed, and then only when the gap is large enough to stand out from luck. The same is done for the stock ranker's own top-5 (`data/paper_model.csv`). A "Growth score check" on the Trust tab shows, over time, which of the four parts actually ranked stocks well.

**It never trades.** There is no broker connection, key or order code anywhere in this project. Before anything like real trading is considered, the paper record should be long and clearly positive after costs, and you should expect a weekly turnover this high to cost real money.

Turn it off with `GROWTH_ENABLED = False` or `PAPER_ENABLED = False` (in `marketlab/config.py`), or run `python run_weekly.py --no-growth --no-paper`. The number of picks, the tilt and the cap are `GROWTH_TOP_N`, `GROWTH_TILT` and `GROWTH_MAX_W` in `marketlab/opts.py`.

## The watchlist
The stocks the app follows are listed in `watchlist.txt` in the main folder (about 110 US, UK, European and Japanese shares). Edit it in the GitHub app (open the file, tap the pencil): one ticker per line, `#` starts a comment, UK shares end in `.L` (for example `BARC.L`), class shares use a dash (`BRK-B`). Between 14 and 120 tickers are used; if the file is missing or unusable the built-in list is used instead and the page says so.
- Each stock's market (and so its risk scaler) is picked from its ending: `.L` is the FTSE 100, most other European endings the Euro Stoxx 50, `.T` the Nikkei 225, and everything else the S&P 500.
- If you change the list a lot, the next run redoes the walk-forward model check, so it takes longer. History in the logs and paper ledgers is kept, and any stock held in the paper portfolio is still priced after you remove it.
- A misspelt ticker is skipped with a warning. The run stops with a clear message only if almost nothing can be downloaded.

## Global drivers
Besides the original rates, dollar, gold, oil, copper, credit and fear readings, the app now also tracks 5- and 30-year yields, 3-month volatility, Brent crude, natural gas, silver, the euro, pound and yen against the dollar, long bonds, emerging markets, China, semiconductors, small companies and the equal-weight S&P 500 (all optional; one that cannot be downloaded is simply left out).
1. **Readings.** The Markets tab shows each driver's level, its recent move and whether that move is unusual against its own past year.
2. **Inputs.** Each driver's recent move can be given to the market and stock models as extra inputs. Every month the health check runs the model twice, once with and once without them, year by year on earlier data only, and uses the extra inputs **only if they clearly help** (a gain in most years, and overall). On random data the test correctly says no. When the answer is no, the drivers are still shown, but the models do not use them, and the Trust tab says so.
3. **Stock sensitivities.** For each stock, the app measures how it has typically reacted to rates, the dollar, oil, credit and market fear over the past six months, and combines that with the driver's recent move. Where a stock is clearly exposed, the Growth watch gets a tag such as "helped by rising oil". These are tags for you to read; they do not change the score.

Turn them off with `MACRO_ENABLED = False`, or `python run_weekly.py --no-macro`.

## Changing settings
Edit `marketlab/config.py` (you can do it in the GitHub app: open the file and tap the pencil). It holds the stock lists, horizons, number of picks, volatility target and costs. The tests run automatically when code changes, and a green tick on the Tests workflow means the edit did not break anything. To change the schedule, edit the `cron` line in `.github/workflows/weekly.yml`.

## Updating
Run the deploy notebook again. It overwrites the code files but **keeps** anything that holds your data or choices:
- `data/` (your forecast logs, health file, news archive, company-data history and paper-trading ledgers) and `docs/data/latest.json` are never overwritten.
- `watchlist.txt` is kept if it already exists, so your own edits are safe.
- `marketlab/config.py` is kept if you changed it. If a new version needs new settings the notebook tells you. Set `REPLACE_CONFIG = True` in its first cell to replace it with the new default.

## Running it on a computer
```
pip install -r requirements-dev.txt
pytest -q                        # tests, using synthetic data (no network needed)
python run_weekly.py             # a normal weekly run
python run_weekly.py --refresh-health   # also force the walk-forward health check now
python run_weekly.py --no-outlook --no-headlines
python run_weekly.py --no-news --no-fundamentals   # skip the news archive and company data
python run_weekly.py --no-growth --no-paper --no-macro   # skip the Growth watch, paper trading and global drivers
```

## Things that can go wrong
- **The run fails:** Yahoo Finance data (via the unofficial `yfinance` library) sometimes blocks or rate-limits cloud servers. The job retries, and refuses to publish anything if the data is incomplete or stale, so a bad run never pollutes your log. Re-run it from the Actions tab. If it keeps failing, a paid, keyed data provider is the dependable fix.
- **News tone or company data is missing:** GDELT or Yahoo can refuse a request. The page says which part is unavailable and the next run tries again. Company data is only saved for weeks it could be fetched, so a gap just means a missing week.
- **The first run after this update is slower:** the monthly model check is redone for the larger watchlist and the global-driver test is added. A normal week afterwards is much quicker. The weekly job is allowed up to two hours.
- **The Growth tab says "unavailable":** it needs at least 25 stocks with prices that week. Check `watchlist.txt` for typos.
- **Scheduled runs stop:** GitHub can disable scheduled workflows in a repository that has had no activity for a long time. If the dashboard shows an "old data" banner, check the Actions tab and re-enable the workflow. Running it by hand also counts as activity.
- **Dashboard looks stale after an update:** close the app fully and reopen it.

## Privacy note
Anything in a public repository, including your code, forecast log and the dashboard page, is visible to anyone who finds it. A GitHub Pages site is reachable by anyone with the link. That is fine for a personal experiment, but it is not a place for anything you would not want shown publicly. Never put API keys or tokens in the code or settings file; use repository secrets.

## Disclaimer
This is research software for learning and experimentation. It is not investment advice, it comes with no guarantee of accuracy, and past results do not predict future returns. Do not trade real money on it without understanding the risks, and never risk money you cannot afford to lose.
