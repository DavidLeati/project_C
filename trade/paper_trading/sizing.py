from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, ROUND_DOWN


@dataclass(frozen=True)
class SymbolFilters:
    quantity_step: Decimal
    min_quantity: Decimal
    min_notional: Decimal

    @classmethod
    def from_exchange_symbol(cls, symbol_info: dict) -> "SymbolFilters":
        lot_filter = next(f for f in symbol_info["filters"] if f["filterType"] in {"LOT_SIZE", "MARKET_LOT_SIZE"})
        notional_filter = next(
            (f for f in symbol_info["filters"] if f["filterType"] in {"MIN_NOTIONAL", "NOTIONAL"}),
            {},
        )
        return cls(
            quantity_step=Decimal(str(lot_filter["stepSize"])),
            min_quantity=Decimal(str(lot_filter["minQty"])),
            min_notional=Decimal(str(notional_filter.get("notional", notional_filter.get("minNotional", "0")))),
        )


@dataclass(frozen=True)
class PositionPlan:
    side: str | None
    quantity: Decimal
    target_notional: Decimal
    current_notional: Decimal
    delta_notional: Decimal
    reduce_only: bool
    reason: str

    @property
    def should_order(self) -> bool:
        return self.side is not None and self.quantity > 0


def _floor_to_step(value: Decimal, step: Decimal) -> Decimal:
    if step <= 0:
        return value
    return (value / step).to_integral_value(rounding=ROUND_DOWN) * step


def plan_position_order(
    *,
    model_position: float,
    margin_balance: Decimal,
    mark_price: Decimal,
    current_position_qty: Decimal,
    filters: SymbolFilters,
) -> PositionPlan:
    # Allow up to 20x leverage 
    bounded_position = max(-20.0, min(20.0, float(model_position)))
    target_notional = margin_balance * Decimal(str(bounded_position))
    current_notional = current_position_qty * mark_price
    delta_notional = target_notional - current_notional

    if abs(delta_notional) < filters.min_notional:
        return PositionPlan(None, Decimal("0"), target_notional, current_notional, delta_notional, False, "below_min_notional")

    raw_qty = abs(delta_notional) / mark_price
    quantity = _floor_to_step(raw_qty, filters.quantity_step)
    if quantity < filters.min_quantity:
        return PositionPlan(None, Decimal("0"), target_notional, current_notional, delta_notional, False, "below_min_quantity")

    side = "BUY" if delta_notional > 0 else "SELL"
    reduce_only = (
        current_position_qty > 0 and side == "SELL" and target_notional >= 0
    ) or (
        current_position_qty < 0 and side == "BUY" and target_notional <= 0
    )
    return PositionPlan(side, quantity, target_notional, current_notional, delta_notional, reduce_only, "order")
