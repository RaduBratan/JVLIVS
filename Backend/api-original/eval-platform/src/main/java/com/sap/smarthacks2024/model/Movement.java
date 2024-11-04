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
@Table(name = "requested_movement")
public class Movement {

	@Id
	@GeneratedValue(strategy = GenerationType.UUID)
	@Column(name = "id", nullable = false, updatable = false)
	private UUID id;

	@ManyToOne(optional = false)
	@JoinColumn(name = "session_id")
	private EvaluationSession session;

	@ManyToOne(optional = false)
	@JoinColumn(name = "connection_id")
	private Connection connection;
	
	@ManyToOne(optional = false)
	@JoinColumn(name = "from_id")
	private NodeStatus from;
	
	@ManyToOne(optional = false)
	@JoinColumn(name = "to_id")
	private NodeStatus to;
	

	@Column(name = "amount")
	private long amount;

	@Column(name = "day_posted")
	private int dayPosted;

	@Column(name = "day_delivered")
	private int dayDelivered;

	@Column(name = "cost")
	private Double cost;

	@Column(name = "co2")
	private Double co2;

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

	public Connection getConnection() {
		return connection;
	}

	public void setConnection(Connection connection) {
		this.connection = connection;
	}

	public long getAmount() {
		return amount;
	}

	public void setAmount(long amount) {
		this.amount = amount;
	}

	public int getDayPosted() {
		return dayPosted;
	}

	public void setDayPosted(int dayPosted) {
		this.dayPosted = dayPosted;
	}

	public int getDayDelivered() {
		return dayDelivered;
	}

	public void setDayDelivered(int dayDelivered) {
		this.dayDelivered = dayDelivered;
	}

	public Double getCost() {
		return cost;
	}

	public void setCost(Double cost) {
		this.cost = cost;
	}

	public Double getCo2() {
		return co2;
	}

	public void setCo2(Double co2) {
		this.co2 = co2;
	}

	public NodeStatus getFrom() {
		return from;
	}

	public void setFrom(NodeStatus from) {
		this.from = from;
	}

	public NodeStatus getTo() {
		return to;
	}

	public void setTo(NodeStatus to) {
		this.to = to;
	}

}
