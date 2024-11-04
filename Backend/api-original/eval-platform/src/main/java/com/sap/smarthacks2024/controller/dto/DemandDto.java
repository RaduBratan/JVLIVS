package com.sap.smarthacks2024.controller.dto;

import java.util.UUID;

import io.swagger.v3.oas.annotations.media.Schema;

@Schema(name = "Demand", description = "The demand / order of a customer")
public record DemandDto(@Schema(description = "The ID of the customer") UUID customerId,
		@Schema(description = "The amount requested") long amount,
		@Schema(description = "The day when the demand was posted") int postDay,
		@Schema(description = "Delivery should happen no earlier than this day") int startDay,
		@Schema(description = "Delivery should happen no later than this day") int endDay) {

}
