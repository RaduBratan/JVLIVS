package com.sap.smarthacks2024.controller.dto;

import java.util.List;

import com.fasterxml.jackson.annotation.JsonInclude;

import io.swagger.v3.oas.annotations.media.Schema;

@Schema(name = "DayResponse", description = "The success response to a round")
@JsonInclude(JsonInclude.Include.NON_NULL)
public record DayResponseDto(@Schema(description = "The current day / round") int round,
		@Schema(description = "The list of posted customer demands (orders) on the current day. It can be empty or missing.") List<DemandDto> demand,
		@Schema(description = "The list of generated penalties as a result of running the day / round. Penalties can be also generated for the wrong state of the network.") List<PenaltyDto> penalties,
		@Schema(description = "KPIs changes for the current day / round") KpisDto deltaKpis,
		@Schema(description = "Cummulative KPIs (round to day)") KpisDto totalKpis) {

}
