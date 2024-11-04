package com.sap.smarthacks2024.controller.dto;

import io.swagger.v3.oas.annotations.media.Schema;

@Schema(name = "Kpis", description = "The KPIs that are the metrics of the simulation")
public record KpisDto(@Schema(description = "The reference day") int day, @Schema(description = "the cost") double cost,
		@Schema(description = "the co2 value") double co2) {

}
