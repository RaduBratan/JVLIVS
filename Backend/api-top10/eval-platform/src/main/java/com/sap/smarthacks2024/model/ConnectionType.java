package com.sap.smarthacks2024.model;

public enum ConnectionType {
	
	PIPELINE(0.05d,0.02d,1.13d),
	TRUCK(0.42d,0.31d,0.73d);
	
	private final Double costPerDistanceAndVolume;
	private final Double co2PerDistanceAndVolume;
	private final Double overUsePenaltyPerVolume;
	
	ConnectionType(Double costPerDistanceAndVolume, Double co2PerDistanceAndVolume, Double overUsePenaltyPerVolume) {
		this.costPerDistanceAndVolume = costPerDistanceAndVolume;
		this.co2PerDistanceAndVolume = co2PerDistanceAndVolume;
		this.overUsePenaltyPerVolume = overUsePenaltyPerVolume;
	}
	
	public Double getCostPerDistanceAndVolume() {
		return costPerDistanceAndVolume;
	}
	
	public Double getCo2PerDistanceAndVolume() {
		return co2PerDistanceAndVolume;
	}
	
	public Double getOverUsePenaltyPerVolume() {
		return overUsePenaltyPerVolume;
	}
}
