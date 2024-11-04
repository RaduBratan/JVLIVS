package com.sap.smarthacks2024.model;

import java.time.LocalDateTime;
import java.util.UUID;

import jakarta.persistence.*;

@Entity
@Table(name = "eval_track")
public class EvaluationTrack {

	@Id
	@GeneratedValue(strategy = GenerationType.UUID)
	@Column(name = "id", nullable = false, updatable = false, insertable = false)
	private UUID id;

	@ManyToOne(optional = false)
	@JoinColumn(name = "team_id")
	private Team team;

	@Column(name = "prod_Day")
	private int prodDay;

	@Column(name = "production_Cost")
	private double productionCost;

	@Column(name = "production_Co2")
	private double productionCo2;

	@Column(name = "movement_Cost")
	private double movementCost;

	@Column(name = "movement_Co2")
	private double movementCo2;

	@Column(name = "penalty_Cost")
	private double penaltyCost;

	@Column(name = "penalty_Co2")
	private double penaltyCo2;

	@Column(name = "latest")
	private boolean latest;

	@Column(name = "session_id")
	private UUID sessionId;

	@Temporal(TemporalType.TIMESTAMP)
	@Column(name = "time_received")
	private LocalDateTime timeReceived;

	public boolean getLatest() {
		return latest;
	}

	public void setLatest(boolean latest) {
		this.latest = latest;
	}

	public UUID getSessionId() {
		return sessionId;
	}

	public void setSessionId(UUID sessionId) {
		this.sessionId = sessionId;
	}

	public double getPenaltyCo2() {
		return penaltyCo2;
	}

	public void setPenaltyCo2(double penaltyCo2) {
		this.penaltyCo2 = penaltyCo2;
	}

	public UUID getId() {
		return id;
	}

	public void setId(UUID id) {
		this.id = id;
	}

	public double getProductionCost() {
		return productionCost;
	}

	public void setProductionCost(double productionCost) {
		this.productionCost = productionCost;
	}

	public double getProductionCo2() {
		return productionCo2;
	}

	public void setProductionCo2(double productionCo2) {
		this.productionCo2 = productionCo2;
	}

	public double getMovementCost() {
		return movementCost;
	}

	public void setMovementCost(double movementCost) {
		this.movementCost = movementCost;
	}

	public double getMovementCo2() {
		return movementCo2;
	}

	public void setMovementCo2(double movementCo2) {
		this.movementCo2 = movementCo2;
	}

	public double getPenaltyCost() {
		return penaltyCost;
	}

	public void setPenaltyCost(double penaltyCost) {
		this.penaltyCost = penaltyCost;
	}

	public int getProdDay() {
		return prodDay;
	}

	public void setProdDay(int prodDay) {
		this.prodDay = prodDay;
	}

	public LocalDateTime getTimeReceived() {
		return timeReceived;
	}

	public void setTimeReceived(LocalDateTime timeReceived) {
		this.timeReceived = timeReceived;
	}

    public Team getTeam() {
        return team;
    }

    public void setTeam(Team team) {
        this.team = team;
    }
}
