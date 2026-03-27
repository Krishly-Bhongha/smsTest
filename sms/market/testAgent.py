from typing import Dict, List, Optional
from sim.agents.strategy_interface import Strategy, Observation, OrderRequest
from sim.exchange import OrderType, Side
from sim.news import NewsEvent


class MyStrategy(Strategy):
    def __init__(self, seed=None):
        # --- Tunable parameters ---
        self.spread_width = 2.0        # half-spread added each side of mid
        self.order_size = 10.0         # base quantity per order
        self.skew_factor = 0.05        # how aggressively to skew quotes per unit of position
        self.max_position = 100.0      # hard position limit per commodity
        self.news_bias_strength = 3.0  # initial price shift from news
        self.news_decay = 0.85         # multiplicative decay per tick for news bias
        self.volatility_spread_mult = 1.5  # widen spread when volatility is high
        self.vol_lookback = 5          # number of recent trades to gauge volatility

        # --- Internal state ---
        self.news_bias: Dict[str, float] = {}        # commodity -> directional bias
        self.last_midprices: Dict[str, List[float]] = {}  # for simple vol estimate

    # --------------------------------------------------------------------- #
    #  Core logic – called every tick
    # --------------------------------------------------------------------- #
    def act(self, observations: Dict[str, Observation]) -> List[OrderRequest]:
        orders: List[OrderRequest] = []

        for commodity, obs in observations.items():
            # ---- 1. Determine a fair price to quote around ----
            mid = self._fair_price(commodity, obs)
            if mid is None:
                continue  # no price info at all – skip

            # Track midprices for volatility estimate
            self.last_midprices.setdefault(commodity, []).append(mid)
            if len(self.last_midprices[commodity]) > self.vol_lookback + 1:
                self.last_midprices[commodity] = self.last_midprices[commodity][-(self.vol_lookback + 1):]

            # ---- 2. Compute dynamic half-spread ----
            half = self._dynamic_half_spread(commodity, obs)

            # ---- 3. Inventory skew ----
            skew = -obs.position * self.skew_factor

            # ---- 4. News bias (decays each tick) ----
            bias = self.news_bias.get(commodity, 0.0)
            self.news_bias[commodity] = bias * self.news_decay  # decay for next tick

            # ---- 5. Compute quote prices ----
            bid_price = round(mid + skew + bias - half, 2)
            ask_price = round(mid + skew + bias + half, 2)

            # ---- 6. Size adjustment near position limits ----
            bid_qty, ask_qty = self._adjusted_sizes(obs.position)

            # ---- 7. Place orders ----
            if bid_qty > 0:
                orders.append(
                    OrderRequest(
                        side=Side.BID,
                        order_type=OrderType.LIMIT,
                        quantity=bid_qty,
                        commodity=commodity,
                        price=bid_price,
                    )
                )
            if ask_qty > 0:
                orders.append(
                    OrderRequest(
                        side=Side.ASK,
                        order_type=OrderType.LIMIT,
                        quantity=ask_qty,
                        commodity=commodity,
                        price=ask_price,
                    )
                )

            # ---- 8. Aggressive inventory unwind if way over limit ----
            if abs(obs.position) > self.max_position * 0.9:
                orders.extend(self._unwind_orders(commodity, obs))

        return orders

    # --------------------------------------------------------------------- #
    #  News handler – fires instantly on broadcast
    # --------------------------------------------------------------------- #
    def on_news(self, news: Optional[NewsEvent]) -> None:
        if news is None:
            return

        direction = self._interpret_news(news)
        commodity = self._news_commodity(news)
        if commodity is not None:
            self.news_bias[commodity] = self.news_bias.get(commodity, 0.0) + direction * self.news_bias_strength

    # --------------------------------------------------------------------- #
    #  Housekeeping
    # --------------------------------------------------------------------- #
    def reset(self) -> None:
        self.news_bias.clear()
        self.last_midprices.clear()

    def refresh_orders(self) -> bool:
        # Cancel and requote every tick – essential for market making
        return True

    # ================================================================== #
    #  Private helpers
    # ================================================================== #

    def _fair_price(self, commodity: str, obs: Observation) -> Optional[float]:
        """Best estimate of current fair value."""
        if obs.midprice is not None:
            return obs.midprice
        if obs.reference_price is not None:
            return obs.reference_price
        # Fall back to last trade price
        if obs.last_trades:
            return obs.last_trades[-1][0]
        return None

    def _dynamic_half_spread(self, commodity: str, obs: Observation) -> float:
        """Widen the spread when recent price moves are large."""
        base = self.spread_width

        # Simple realised-vol proxy: average absolute tick-to-tick change
        prices = self.last_midprices.get(commodity, [])
        if len(prices) >= 2:
            changes = [abs(prices[i] - prices[i - 1]) for i in range(1, len(prices))]
            avg_change = sum(changes) / len(changes)
            if avg_change > base * 0.5:
                base = max(base, avg_change * self.volatility_spread_mult)

        # Also respect the current book spread – don't quote inside it too far
        if obs.spread is not None and obs.spread > 0:
            base = max(base, obs.spread * 0.4)

        return base

    def _adjusted_sizes(self, position: float) -> tuple:
        """Reduce size on the side that would increase an already-large position."""
        bid_qty = self.order_size
        ask_qty = self.order_size

        ratio = abs(position) / self.max_position if self.max_position else 0.0

        if position > 0:
            # Long – reduce bids, boost asks
            bid_qty *= max(0.0, 1.0 - ratio)
            ask_qty *= min(2.0, 1.0 + ratio)
        elif position < 0:
            # Short – reduce asks, boost bids
            ask_qty *= max(0.0, 1.0 - ratio)
            bid_qty *= min(2.0, 1.0 + ratio)

        return round(bid_qty, 2), round(ask_qty, 2)

    def _unwind_orders(self, commodity: str, obs: Observation) -> List[OrderRequest]:
        """Fire a market order to reduce a dangerously large position."""
        unwind: List[OrderRequest] = []
        excess = abs(obs.position) - self.max_position * 0.85
        if excess <= 0:
            return unwind

        qty = round(min(excess, self.order_size * 2), 2)
        side = Side.ASK if obs.position > 0 else Side.BID

        unwind.append(
            OrderRequest(
                side=side,
                order_type=OrderType.MARKET,
                quantity=qty,
                commodity=commodity,
            )
        )
        return unwind

    def _interpret_news(self, news: NewsEvent) -> float:
        """
        Return +1 for bullish, -1 for bearish, 0 for neutral.
        Extend this with NLP or keyword matching for richer signals.
        """
        text = str(news).lower()
        bullish_kw = ["surge", "rise", "bullish", "upgrade", "positive", "growth",
                       "rally", "strong", "beat", "above", "boom", "up"]
        bearish_kw = ["drop", "fall", "bearish", "downgrade", "negative", "decline",
                       "crash", "weak", "miss", "below", "slump", "down"]

        score = 0.0
        for w in bullish_kw:
            if w in text:
                score += 1.0
        for w in bearish_kw:
            if w in text:
                score -= 1.0

        if score > 0:
            return 1.0
        elif score < 0:
            return -1.0
        return 0.0

    def _news_commodity(self, news: NewsEvent) -> Optional[str]:
        """Extract which commodity the news applies to."""
        if hasattr(news, "commodity") and news.commodity:
            return news.commodity
        return None