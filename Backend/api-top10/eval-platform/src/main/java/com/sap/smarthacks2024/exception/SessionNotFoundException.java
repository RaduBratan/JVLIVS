package com.sap.smarthacks2024.exception;

import org.springframework.http.HttpStatus;

public class SessionNotFoundException extends BusinessException {

	private static final long serialVersionUID = 1L;

	public SessionNotFoundException() {
		super(HttpStatus.NOT_FOUND, "SESS-003", "Session not found");
	}

}
