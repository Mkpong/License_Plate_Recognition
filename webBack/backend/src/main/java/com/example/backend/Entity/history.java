package com.example.backend.Entity;

import jakarta.persistence.*;
import lombok.Data;

@Entity
@Data
@Table(name="recode")
public class history {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private int id;

    @Column(name= "carNumber")
    private String carNumber;

    @Column(name="enterTime")
    private String entranceTime;

    @Column(name="departureTime")
    private String departureTime;

    @Column(name = "parkingFee")
    private int parkingFee;

    @Column(name = "plateType")
    private int plateType;
}