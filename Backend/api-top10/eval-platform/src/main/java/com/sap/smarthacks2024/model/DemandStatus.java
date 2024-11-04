package com.sap.smarthacks2024.model;

import java.util.UUID;

import jakarta.persistence.Column;
import jakarta.persistence.Entity;
import jakarta.persistence.GeneratedValue;
import jakarta.persistence.GenerationType;
import jakarta.persistence.Id;
import jakarta.persistence.JoinColumn;
import jakarta.persistence.ManyToOne;
import jakarta.persistence.Table;

@Entity
@Table(name = "demand_status")
public class DemandStatus {
	@Id
	@GeneratedValue(strategy = GenerationType.UUID)
	@Column(name = "id", nullable = false, updatable = false)
	private UUID id;

	@ManyToOne(optional = false)
	@JoinColumn(name = "session_id")
	private EvaluationSession session;

	@ManyToOne
	@JoinColumn(name = "demand_id")
	private Demand demand;
	
	@ManyToOne
	@JoinColumn(name = "node_status_id")
	private NodeStatus customerNodeStatus;

	@Column(name = "remaining_quantity")
	private long remainingQuantity;

	public UUID getId() {
		return id;
	}

	public void setId(UUID id) {
		this.id = id;
	}

	public long getRemainingQuantity() {
		return remainingQuantity;
	}

	public void setRemainingQuantity(long remainingQuantity) {
		this.remainingQuantity = remainingQuantity;
	}

	public EvaluationSession getSession() {
		return session;
	}

	public void setSession(EvaluationSession session) {
		this.session = session;
	}

	public Demand getDemand() {
		return demand;
	}

	public void setDemand(Demand demand) {
		this.demand = demand;
	}

	public NodeStatus getCustomerNodeStatus() {
		return customerNodeStatus;
	}

	public void setCustomerNodeStatus(NodeStatus customerNodeStatus) {
		this.customerNodeStatus = customerNodeStatus;
	}

}
