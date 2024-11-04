package com.sap.smarthacks2024.model;

import jakarta.persistence.Column;
import jakarta.persistence.DiscriminatorValue;
import jakarta.persistence.Entity;

@Entity
@DiscriminatorValue("REFINERY")
public class Refinery extends Node {
	@Column(name = "capacity")
	private Long capacity;

	@Column(name = "max_output")
	private Long maxOutput;

	@Column(name = "production")
	private Long production;

	@Column(name = "overflow_penalty")
	private Double overflowPenaltyCoefficient;

	@Column(name = "underflow_penalty")
	private Double underflowPenaltyCoefficient;

	@Column(name = "over_output_penalty")
	private Double overOutputPenaltyCoefficient;

	@Column(name = "production_cost")
	private Double productionCostPerUnit;

	@Column(name = "production_co2")
	private Double productionCO2PerUnit;

	@Column(name = "initial_stock")
	private Long initialStock;

	public Long getCapacity() {
		return capacity;
	}

	public void setCapacity(Long capacity) {
		this.capacity = capacity;
	}

	public Long getMaxOutput() {
		return maxOutput;
	}

	public void setMaxOutput(Long maxOutput) {
		this.maxOutput = maxOutput;
	}

	public Long getProduction() {
		return production;
	}

	public void setProduction(Long production) {
		this.production = production;
	}

	public Double getOverflowPenaltyCoefficient() {
		return overflowPenaltyCoefficient;
	}

	public void setOverflowPenaltyCoefficient(Double overflowPenaltyCoefficient) {
		this.overflowPenaltyCoefficient = overflowPenaltyCoefficient;
	}

	public Double getUnderflowPenaltyCoefficient() {
		return underflowPenaltyCoefficient;
	}

	public void setUnderflowPenaltyCoefficient(Double underflowPenaltyCoefficient) {
		this.underflowPenaltyCoefficient = underflowPenaltyCoefficient;
	}

	public Double getOverOutputPenaltyCoefficient() {
		return overOutputPenaltyCoefficient;
	}

	public void setOverOutputPenaltyCoefficient(Double overOutputPenaltyCoefficient) {
		this.overOutputPenaltyCoefficient = overOutputPenaltyCoefficient;
	}

	public Double getProductionCostPerUnit() {
		return productionCostPerUnit;
	}

	public void setProductionCostPerUnit(Double productionCostPerUnit) {
		this.productionCostPerUnit = productionCostPerUnit;
	}

	public Double getProductionCO2PerUnit() {
		return productionCO2PerUnit;
	}

	public void setProductionCO2PerUnit(Double productionCO2PerUnit) {
		this.productionCO2PerUnit = productionCO2PerUnit;
	}

	public Long getInitialStock() {
		return initialStock;
	}

	public void setInitialStock(Long initialStock) {
		this.initialStock = initialStock;
	}
}
