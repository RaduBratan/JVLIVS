package com.sap.smarthacks2024.controller.dto;

import com.fasterxml.jackson.annotation.JsonInclude;

import java.util.UUID;

@JsonInclude(JsonInclude.Include.NON_NULL)
public record EvalTrackDto(UUID teamName
        , int day
        , double productionCost, double productionCo2
        , double moveCost, double moveCo2
        , double penaltyCost, double penaltyCo2
        , long score) {
}
