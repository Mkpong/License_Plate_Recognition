package com.example.backend.DTO;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;
import lombok.ToString;

@Data
@ToString
@AllArgsConstructor
@NoArgsConstructor
public class carDataDTO {

    private String state;

    private String ticket;

    private int parkingFee;

    private String parkingTime;

}
