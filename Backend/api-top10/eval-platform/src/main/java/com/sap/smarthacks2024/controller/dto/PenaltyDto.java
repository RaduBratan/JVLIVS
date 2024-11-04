package com.sap.smarthacks2024.controller.dto;

import com.fasterxml.jackson.annotation.JsonInclude;

import io.swagger.v3.oas.annotations.media.Schema;

@JsonInclude(JsonInclude.Include.NON_NULL)
@Schema(name = "Penalty", description = "Penalty generated as a result of running the day / round")
public record PenaltyDto(@Schema(description = "The day when the penalty was issued") int day,
		@Schema(description = "The type of the penalty", allowableValues = {
				"INVALID_CONNECTION", "REFINERY_OVER_OUTPUT", "STORAGE_TANK_OVER_OUTPUT", "STORAGE_TANK_OVER_INPUT",
				"CUSTOMER_OVER_INPUT", "CONNECTION_OVER_CAPACITY", "CUSTOMER_UNEXPECTED_DELIVERY",
				"CUSTOMER_EARLY_DELIVERY", "CUSTOMER_LATE_DELIVERY", "REFINERY_OVERFLOW", "STORAGE_TANK_OVERFLOW",
				"REFINERY_UNDERFLOW", "STORAGE_TANK_UNDERFLOW", "PENDING_MOVEMENTS", "UNMET_DEMANDS" }) String type,
		@Schema(description = "The message explaining the penalty") String message,
		@Schema(description = "The applied cost of the penalty") Double cost,
		@Schema(description = "The applied co2 of the penalty") Double co2){

}
