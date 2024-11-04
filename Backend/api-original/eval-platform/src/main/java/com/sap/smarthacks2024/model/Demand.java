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
@Table(name = "demand")
public class Demand {
	@Id
	@GeneratedValue(strategy = GenerationType.UUID)
	@Column(name = "id", nullable = false, updatable = false)
	private UUID id;

	@ManyToOne(optional = false)
	@JoinColumn(name = "customer_id")
	private Customer customer;

	@Column(name = "quantity")
	private long quantity;

	@Column(name = "post_day")
	private int postDay;

	@Column(name = "start_delivery_day")
	private int startDeliveryDay;

	@Column(name = "end_delivery_day")
	private int endDeliveryDay;

	public UUID getId() {
		return id;
	}

	public void setId(UUID id) {
		this.id = id;
	}

	public Customer getCustomer() {
		return customer;
	}

	public void setCustomer(Customer customer) {
		this.customer = customer;
	}

	public long getQuantity() {
		return quantity;
	}

	public void setQuantity(long quantity) {
		this.quantity = quantity;
	}

	public int getPostDay() {
		return postDay;
	}

	public void setPostDay(int postDay) {
		this.postDay = postDay;
	}

	public int getStartDeliveryDay() {
		return startDeliveryDay;
	}

	public void setStartDeliveryDay(int startDeliveryDay) {
		this.startDeliveryDay = startDeliveryDay;
	}

	public int getEndDeliveryDay() {
		return endDeliveryDay;
	}

	public void setEndDeliveryDay(int endDeliveryDay) {
		this.endDeliveryDay = endDeliveryDay;
	}

}
