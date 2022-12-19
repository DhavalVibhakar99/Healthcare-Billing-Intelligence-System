SELECT AVG(treatment.Total) AS AVG_of_InvoiceTotal, physician.Phy_specs
FROM invoices
JOIN treatment
ON invoices.treat_id = treatment.TreatID
JOIN physician
ON invoices.phy_id = physician.phyID
JOIN patient
ON invoices.p_id=patient.patID
GROUP BY Phy_specs