package com.example.backend.Controller;

import com.example.backend.DTO.SeasonTicketDTO;
import com.example.backend.Entity.history;
import com.example.backend.Entity.parkingCar;
import com.example.backend.Entity.seasonTicketCar;
import com.example.backend.Service.SeasonService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.data.domain.Page;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@RestController
@RequestMapping("/api")
public class SeasonController {

    @Autowired
    SeasonService seasonservice;

    @PostMapping("/season/new")
    public String newSeasonCar(@RequestBody SeasonTicketDTO st){
        return seasonservice.newSeasonCar(st);
    }

    @GetMapping("/season/list")
    public List<seasonTicketCar> getSeasonList(){
        return seasonservice.getSeasonList();
    }

    @PostMapping("/season/search")
    public Page<seasonTicketCar> searchHistory(@RequestParam(value="page", defaultValue = "0") int page,
                                       @RequestParam(value="keyword") String keyword){
        return seasonservice.searchSeasonCar(page, keyword);
    }

}
