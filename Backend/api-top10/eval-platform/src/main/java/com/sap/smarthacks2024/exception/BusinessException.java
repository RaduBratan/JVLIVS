package com.sap.smarthacks2024.exception;

import org.springframework.http.HttpStatusCode;

public class BusinessException extends RuntimeException {

	private static final long serialVersionUID = 1L;
	private final String code;
	private final HttpStatusCode statusCode;

	public BusinessException(HttpStatusCode statusCode, String code, String message) {
		super(message);
		this.code = code;
		this.statusCode = statusCode;
	}

	public String getCode() {
		return code;
	}
	
	public HttpStatusCode getStatusCode() {
		return statusCode;
	}

}
