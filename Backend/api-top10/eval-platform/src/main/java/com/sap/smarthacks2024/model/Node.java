package com.sap.smarthacks2024.model;

import java.util.List;
import java.util.UUID;

import jakarta.persistence.Column;
import jakarta.persistence.DiscriminatorColumn;
import jakarta.persistence.Entity;
import jakarta.persistence.GeneratedValue;
import jakarta.persistence.GenerationType;
import jakarta.persistence.Id;
import jakarta.persistence.Inheritance;
import jakarta.persistence.InheritanceType;
import jakarta.persistence.OneToMany;
import jakarta.persistence.Table;

@Entity
@Table(name = "network_node")
@Inheritance(strategy = InheritanceType.SINGLE_TABLE)
@DiscriminatorColumn(name = "node_type")
public abstract class Node {

	@Id
	@GeneratedValue(strategy = GenerationType.UUID)
	@Column(name = "id", nullable = false, updatable = false)
	private UUID id;

	@Column(name = "name", nullable = false)
	private String name;

	@OneToMany(mappedBy = "from")
	private List<Connection> outboundConnections;

	@OneToMany(mappedBy = "to")
	private List<Connection> inboundConnections;

	@Column(name = "node_type", nullable = false, updatable = false, insertable = false)
	private String type;

	public UUID getId() {
		return id;
	}

	public void setId(UUID id) {
		this.id = id;
	}

	public String getName() {
		return name;
	}

	public void setName(String name) {
		this.name = name;
	}

	public List<Connection> getOutboundConnections() {
		return outboundConnections;
	}

	public void setOutboundConnections(List<Connection> outboundConnections) {
		this.outboundConnections = outboundConnections;
	}

	public List<Connection> getInboundConnections() {
		return inboundConnections;
	}

	public void setInboundConnections(List<Connection> inboundConnections) {
		this.inboundConnections = inboundConnections;
	}

	public String getType() {
		return type;
	}

	public void setType(String type) {
		this.type = type;
	}

}
