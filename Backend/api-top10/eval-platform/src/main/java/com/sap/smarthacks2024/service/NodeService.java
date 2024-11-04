package com.sap.smarthacks2024.service;

import java.util.Collection;

import com.sap.smarthacks2024.model.Customer;
import com.sap.smarthacks2024.model.Refinery;
import com.sap.smarthacks2024.model.StorageTank;

public interface NodeService {
	Collection<Refinery> getRefineries();

	Collection<StorageTank> getStorageTanks();

	Collection<Customer> getCustomers();
}
