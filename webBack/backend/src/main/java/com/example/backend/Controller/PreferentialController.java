package com.example.backend.Controller;

import com.example.backend.Entity.PreferentialTreatmentCar;
import com.example.backend.Service.PreferentialService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.data.domain.Page;
import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/api")
public class PreferentialController {
    @Autowired
    PreferentialService preferentialService;

    @GetMapping("/preferential/search")
    public Page<PreferentialTreatmentCar> searchPreferentialCar(@RequestParam(value="page" , defaultValue = "0")int page,
                                                                @RequestParam(value="keyword") String keyword){
        return preferentialService.searchPreferentialCar(page, keyword);
    }

    @PostMapping("/preferential/new")
    public String addPreferentialCar(@RequestParam(value="carnumber")String carNumber){
        return preferentialService.addPreferentialCar(carNumber);
    }


}
