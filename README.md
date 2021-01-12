p = (a ^ b) v (b ^ c) v (a ^ c)


pa = p(a = T) xor p(a = F)
   = (b v (b ^ c) v c) xor (b ^ c)
   = ((b v c) v (b ^ c)) xor (b ^ c)
   = (b v c) xor (b ^ c)
   = ((b v c) ^ ~(b ^ c)) v (~(b v c) ^ (b ^ c))
   = (b v c) ^ (~b v ~c)


a xor b = (a ^ ~b) v (~a ^ b) 