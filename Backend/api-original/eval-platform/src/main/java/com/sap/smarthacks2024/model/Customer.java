package com.sap.smarthacks2024.model;

import jakarta.persistence.Column;
import jakarta.persistence.DiscriminatorValue;
import jakarta.persistence.Entity;

@Entity
@DiscriminatorValue("CUSTOMER")
public class Customer extends Node {

	@Column(name = "max_input")
	private Long maxInput;

	@Column(name = "over_input_penalty")
	private Double overInputPenaltyCoefficient;

	@Column(name = "late_delivery_penalty")
	private Double lateDeliveryPenaltyCoefficient;

	@Column(name = "early_delivery_penalty")
	private Double earlyDeliveryPenaltyCoefficient;

	public Long getMaxInput() {
		return maxInput;
	}

	public void setMaxInput(Long maxInput) {
		this.maxInput = maxInput;
	}

	public Double getOverInputPenaltyCoefficient() {
		return overInputPenaltyCoefficient;
	}

	public void setOverInputPenaltyCoefficient(Double overInputPenaltyCoefficient) {
		this.overInputPenaltyCoefficient = overInputPenaltyCoefficient;
	}

	public Double getLateDeliveryPenaltyCoefficient() {
		return lateDeliveryPenaltyCoefficient;
	}

	public void setLateDeliveryPenaltyCoefficient(Double lateDeliveryPenaltyCoefficient) {
		this.lateDeliveryPenaltyCoefficient = lateDeliveryPenaltyCoefficient;
	}

	public Double getEarlyDeliveryPenaltyCoefficient() {
		return earlyDeliveryPenaltyCoefficient;
	}

	public void setEarlyDeliveryPenaltyCoefficient(Double earlyDeliveryPenaltyCoefficient) {
		this.earlyDeliveryPenaltyCoefficient = earlyDeliveryPenaltyCoefficient;
	}
	
	
}
