package com.sap.smarthacks2024.model;

import jakarta.persistence.Column;
import jakarta.persistence.DiscriminatorValue;
import jakarta.persistence.Entity;

@Entity
@DiscriminatorValue("STORAGE_TANK")
public class StorageTank extends Node {

	@Column(name = "capacity")
	private Long capacity;

	@Column(name = "max_output")
	private Long maxOutput;

	@Column(name = "max_input")
	private Long maxInput;

	@Column(name = "overflow_penalty")
	private Double overflowPenaltyCoefficient;

	@Column(name = "underflow_penalty")
	private Double underflowPenaltyCoefficient;

	@Column(name = "over_input_penalty")
	private Double overInputPenaltyCoefficient;

	@Column(name = "over_output_penalty")
	private Double overOutputPenaltyCoefficient;

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

	public Long getMaxInput() {
		return maxInput;
	}

	public void setMaxInput(Long maxInput) {
		this.maxInput = maxInput;
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

	public Double getOverInputPenaltyCoefficient() {
		return overInputPenaltyCoefficient;
	}

	public void setOverInputPenaltyCoefficient(Double overInputPenaltyCoefficient) {
		this.overInputPenaltyCoefficient = overInputPenaltyCoefficient;
	}

	public Double getOverOutputPenaltyCoefficient() {
		return overOutputPenaltyCoefficient;
	}

	public void setOverOutputPenaltyCoefficient(Double overOutputPenaltyCoefficient) {
		this.overOutputPenaltyCoefficient = overOutputPenaltyCoefficient;
	}

	public Long getInitialStock() {
		return initialStock;
	}

	public void setInitialStock(Long initialStock) {
		this.initialStock = initialStock;
	}
}
