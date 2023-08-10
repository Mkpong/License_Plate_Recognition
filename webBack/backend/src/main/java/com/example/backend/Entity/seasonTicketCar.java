package com.example.backend.Entity;

import jakarta.persistence.*;
import lombok.Data;

@Entity
@Data
@Table(name="seasonTicketCar")
public class seasonTicketCar {
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private int id;

    @Column(name="carNumber")
    private String carNumber;

    @Column(name="validityPeriod")
    private String validDate;

    @Column(name="autoPay")
    private boolean auto_pay;
}