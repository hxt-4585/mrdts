"""保持 Member 位置的非学习参照组件。"""

from methods.contracts import FlightContext, FlightDecision


class StationaryFlight:
    def decide(self, context: FlightContext) -> FlightDecision:
        return FlightDecision({member_id: (0.0, 0.0) for member_id in context.member_ids})
