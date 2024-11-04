package com.sap.smarthacks2024.model;

import java.util.List;
import java.util.UUID;

import jakarta.persistence.CascadeType;
import jakarta.persistence.Column;
import jakarta.persistence.Entity;
import jakarta.persistence.GeneratedValue;
import jakarta.persistence.GenerationType;
import jakarta.persistence.Id;
import jakarta.persistence.JoinColumn;
import jakarta.persistence.ManyToOne;
import jakarta.persistence.OneToMany;
import jakarta.persistence.Table;

@Entity
@Table(name = "node_status")
public class NodeStatus {

	@Id
	@GeneratedValue(strategy = GenerationType.UUID)
	@Column(name = "id", nullable = false, updatable = false)
	private UUID id;

	@ManyToOne(optional = false)
	@JoinColumn(name = "session_id")
	private EvaluationSession session;

	@ManyToOne(optional = false)
	@JoinColumn(name = "node_id")
	private Node node;

	@Column(name = "stock")
	private Long stock;

	@Column(name = "cost")
	private double cost;

	@Column(name = "co2")
	private double co2;

	@OneToMany(mappedBy = "customerNodeStatus", cascade = CascadeType.ALL, orphanRemoval = true)
	private List<DemandStatus> demand;

	public UUID getId() {
		return id;
	}

	public void setId(UUID id) {
		this.id = id;
	}

	public EvaluationSession getSession() {
		return session;
	}

	public void setSession(EvaluationSession session) {
		this.session = session;
	}

	public Node getNode() {
		return node;
	}

	public void setNode(Node node) {
		this.node = node;
	}

	public Long getStock() {
		return stock;
	}

	public void setStock(Long stock) {
		this.stock = stock;
	}

	public double getCost() {
		return cost;
	}

	public void setCost(double cost) {
		this.cost = cost;
	}

	public double getCo2() {
		return co2;
	}

	public void setCo2(double co2) {
		this.co2 = co2;
	}

	public List<DemandStatus> getDemand() {
		return demand;
	}

	public void setDemand(List<DemandStatus> demand) {
		this.demand = demand;
	}

}
