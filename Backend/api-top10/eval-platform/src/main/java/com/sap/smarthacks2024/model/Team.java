package com.sap.smarthacks2024.model;

import java.util.UUID;

import jakarta.persistence.Column;
import jakarta.persistence.Entity;
import jakarta.persistence.GeneratedValue;
import jakarta.persistence.GenerationType;
import jakarta.persistence.Id;
import jakarta.persistence.Table;

@Entity
@Table(name = "team")
public class Team {

	@Id
	@GeneratedValue(strategy = GenerationType.UUID)
	@Column(name = "id", nullable = false, updatable = false, insertable = false)
	private UUID id;

	@Column(name="color", nullable = true, updatable = false, insertable = false)
	private String color;

	@Column(name = "name", nullable = false, updatable = false, insertable = false)
	private String name;

	@Column(name = "api_key", nullable = false, updatable = false, insertable = false)
	private UUID apiKey;

	@Column(name = "internal_use", nullable = false, updatable = false, insertable = false)
	private boolean internalUse;

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

	public UUID getApiKey() {
		return apiKey;
	}

	public void setApiKey(UUID apiKey) {
		this.apiKey = apiKey;
	}

	public boolean getInternalUse() {
		return internalUse;
	}

	public void setInternalUse(boolean internalUse) {
		this.internalUse = internalUse;
	}

    public String getColor() {
        return color;
    }

    public void setColor(String color) {
        this.color = color;
    }
}
