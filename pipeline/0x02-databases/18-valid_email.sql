-- Write a SQL script that creates a trigger that resets the attribute valid_email only when the email has been changed.
DROP TRIGGER IF EXISTS reset_validation;

DELIMITER $$
CREATE TRIGGER reset_validation
       BEFORE UPDATE
       ON `users` FOR EACH ROW
BEGIN
	IF STRCMP(old.email, new.email) <> 0 THEN
	   SET new.valid_email = 0;
	END IF;
END $$

DELIMITER;
