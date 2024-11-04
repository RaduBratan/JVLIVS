package com.sap.smarthacks2024.controller.dto;

import java.util.UUID;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;

import io.swagger.v3.oas.annotations.media.Schema;

@JsonIgnoreProperties(ignoreUnknown = true)
@Schema(name = "Movement", description = "The proposed movement of amount units, using the connection with the given ID")
public record MovementDto(@Schema(description = "The ID of the connection to be used") UUID connectionId,
		@Schema(description = "The number of units to be moved") long amount) {

}
