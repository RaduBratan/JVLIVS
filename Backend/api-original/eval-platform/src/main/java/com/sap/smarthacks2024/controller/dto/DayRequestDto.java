package com.sap.smarthacks2024.controller.dto;

import java.util.List;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;

import io.swagger.v3.oas.annotations.media.Schema;

@JsonIgnoreProperties(ignoreUnknown = true)
@Schema(name = "DayRequest", description = "The request to play a round")
public record DayRequestDto(@Schema(description = "The current day") int day,
		@Schema(description = "The list of proposed movements") List<MovementDto> movements) {

}
