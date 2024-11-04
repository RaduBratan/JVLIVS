package com.sap.smarthacks2024;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

import io.swagger.v3.oas.annotations.OpenAPIDefinition;
import io.swagger.v3.oas.annotations.enums.SecuritySchemeIn;
import io.swagger.v3.oas.annotations.enums.SecuritySchemeType;
import io.swagger.v3.oas.annotations.security.SecurityScheme;

@SpringBootApplication
@OpenAPIDefinition
@SecurityScheme(name = "API key", type = SecuritySchemeType.APIKEY, in = SecuritySchemeIn.HEADER, paramName = "API-KEY")
public class EvalPlatformApplication {

	public static void main(String[] args) {
		SpringApplication.run(EvalPlatformApplication.class, args);
	}

}
