package com.example.backend.Controller;

import com.example.backend.Entity.history;
import com.example.backend.Service.HistoryService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.data.domain.Page;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@RestController
@RequestMapping("/api")
public class HistoryController {

    @Autowired
    HistoryService historyService;
    @GetMapping("/history")
    public List<history> getHistory(){return historyService.getHistory();}

    @PostMapping("/history/search")
    public Page<history> searchHistory(@RequestParam(value="page", defaultValue = "0") int page,
                                       @RequestParam(value="keyword") String keyword){
        return historyService.searchHistory(page, keyword);
    }


}
