SELECT Phy_name,
MIN(no_of_patients) AS minimum_no_visits,
MAX(no_of_patients) AS maximum_no_of_visits
FROM (SELECT COUNT(invoice_id) AS no_of_patients, Phy_name
FROM invoices AS i
JOIN physician AS Ph
ON i.phy_id = Ph.phyID
JOIN patient AS Pt
ON i.p_id = Pt.patID
GROUP BY patID
ORDER BY COUNT(patID)) AS sub
GROUP BY Phy_name;