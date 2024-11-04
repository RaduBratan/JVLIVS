package com.sap.smarthacks2024.exception;

import org.springframework.http.HttpStatus;

public class SessionAlreadyExistsException extends BusinessException {

	private static final long serialVersionUID = 1L;

	public SessionAlreadyExistsException() {
		super(HttpStatus.CONFLICT, "SESS-002", "An active session already exists for this team");
	}

}
