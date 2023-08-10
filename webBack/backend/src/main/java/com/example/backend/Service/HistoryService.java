package com.example.backend.Service;

import com.example.backend.Entity.history;
import com.example.backend.Repository.HistoryRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.PageRequest;
import org.springframework.data.domain.Pageable;
import org.springframework.data.domain.Sort;
import org.springframework.stereotype.Service;

import java.util.ArrayList;
import java.util.List;

@Service
public class HistoryService {

    @Autowired
    HistoryRepository historyRepository;

    public List<history> getHistory(){
        return historyRepository.findAll();
    }

    public Page<history> searchHistory(int page, String keyword){
        List<Sort.Order> sorts = new ArrayList<>();
        sorts.add(Sort.Order.desc("departureTime"));
        Pageable pageable = PageRequest.of(page,10,Sort.by(sorts));
        return historyRepository.findByCarNumberContaining(pageable,keyword);
    }

}
